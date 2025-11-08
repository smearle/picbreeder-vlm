/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.evolution;

import java.util.Collection;

/**
 * A GeneticFactory is responsible for allocating all memory during evolution.
 * <p>
 * Any method prefixed with "Invalid" indicates that the object will be
 * created with the intention of loading via XML later.  These methods are
 * used to continue a saved series.
 * <p>
 * Any method prefixed with "copy" instead of "create" is used during evolution
 * to clone an object, but assign it a new identifier or marking.
 * 
 * @author Nick Beato
 */
public interface GeneticFactory {
	/**
	 * Creates a genome with no links or nodes.  The genome will receive
	 * an invalid identifier.
	 * <p>
	 * This method is invoked to create a dummy genome so that it may be loaded
	 * from XML later.
	 * 
	 * @return The dummy genome
	 */
	public Genome createInvalidGenome();
	
	/**
	 * Creates a copy of the specified genome, giving it a unique identifier
	 * on the current branch.
	 * <p>
	 * This method is invoked during evolution to create offspring.
	 * 
	 * @param genome The parent
	 * @return The copied genome
	 */
	public Genome copyGenome(Genome genome);
	
	/**
	 * Creates a genome using evolution on the parents.
	 * <p>
	 * This method is invoked when restoring a pruned series.
	 * 
	 * @param parents The parent genomes
	 * @return The offspring genome
	 */
	public Genome createGenome(Collection <Genome> parents);
	
	/**
	 * Creates a new series.  The series will not have any generations in it.
	 * <p>
	 * This method is invoked to create a dummy series so that it may be loaded
	 * from XML later.
	 * 
	 * @return The dummy series
	 */
	public Series createInvalidSeries();

	/**
	 * Creates a new series.  The series will contain one generation randomly
	 * generated from the ParameterTable settings.
	 * 
	 * @return The series
	 */
	public Series createRootSeries();
	
	/**
	 * Creates a series, branching from the specified genome.
	 * 
	 * @param branchFrom The genome to branch from
	 * @return The series
	 */
	public Series createBranchSeries(Genome branchFrom);

	/**
	 * Creates an individual using the parameter table to determine the
	 * structure of the genome.  The number of hidden nodes (fully connected)
	 * is specified in the ParameterTable.
	 * <p>
	 * This method should only be invoked to initialize a population when
	 * no saved data or predecessor is known.  In other words, this creates
	 * the root of an entire tree.
	 * 
	 * @return The individual
	 */
	public Individual createRootIndividual();
	
	/**
	 * Creates an individual for loading later on.
	 * 
	 * @return The dummy individual
	 */
	public Individual createInvalidIndividual();
	
	/**
	 * Creates an individual for the given genome.  The genome will
	 * not be modified.  However, modifying the result's genome
	 * will modify the argument genome.
	 * 
	 * @param genome The genome of the individual
	 * @return The individual
	 */
	public Individual createIndividual(Genome genome);
	
	/**
	 * Creates an individual by applying generators to mate the given
	 * parents.  The generators are inherit to the factory's implementation.
	 * 
	 * @param parents The parents of the created offspring
	 * @return The individual
	 */
	public Individual createIndividual(Collection <Genome> parents);
	
	/**
	 * Creates a new generation for a root series.  This generation
	 * will be filled based on values in the ParameterTable.
	 * 
	 * @return The generation
	 */
	public Generation createRootGeneration();
	
	/**
	 * Creates a new generation for a series to load via XML at a later
	 * time.  This generation should be empty.
	 * 
	 * @return The dummy generation
	 */
	public Generation createInvalidGeneration();
	
	/**
	 * Creates a new generation using the specified parents.  Each individual
	 * of the new generation is a decendant of one or more of these parents.
	 * 
	 * @param parents The parents
	 * @param generationNumber The age of this generation
	 * @return The generation
	 */
	public Generation createGeneration(Collection <Genome> parents, int generationNumber);

	/**
	 * Creates a new link, with a unique marking, connecting the source
	 * node to the destination node.
	 * <p>
	 * This method is invoked when a generator wants to add a new link
	 * to a genome during evolution.
	 * 
	 * @param source The source node
	 * @param destination The destination node
	 * @return The link connecting the two nodes
	 */
	public Link createLink(Node source, Node destination);
	
	/**
	 * Creates a new link with an invalid marking.
	 * <p>
	 * This method is invoked to create a dummy link so that it may be loaded
	 * from XML later.
	 * 
	 * @return The dummy link
	 */
	public Link createInvalidLink();

	/**
	 * Copies the given link, giving it a unique marking.
	 * <p>
	 * This method is invoked by generators to create offspring during evolution.
	 * 
	 * @param link The link to copy
	 * @return The copied link
	 */
	public Link copyLink(Link link);

	/**
	 * Creates a new node from the given link.
	 * <p>
	 * This method is invoked by generators who wish to split a link
	 * during evolution.
	 * 
	 * @param link The link to split
	 * @return The node
	 */
	public Node createNode(Link link);
	
	/**
	 * Creates a node with an invalid marking.  This should only be called when
	 * loading a genome.
	 * 
	 * @return The dummy node
	 */
	public Node createInvalidNode();
	
	/**
	 * Creates a node with the given label and type.  This node
	 * will receive a new historical marking.
	 * <p>
	 * This function is only nescessary when creating the root
	 * of a series.  The labels of the nodes are taken from the
	 * parameter table when creating the first generation.
	 * <p>
	 * The labels are only nescessary for the input and output
	 * layers of the network.
	 * 
	 * @param label The label of the node
	 * @param type The type of the node
	 * @return The node
	 */
	public Node createNode(String label, String type);

	/**
	 * Copies the given node, giving it a unique marking.
	 * <p>
	 * This method is invoked by generators to create offspring during evolution.
	 * 
	 * @param node The node to copy
	 * @return The copied node
	 */
	public Node copyNode(Node node);

	/**
	 * Creates a phenotype of an individual.  The phenotype will be updated
	 * via the {@link Phenotype#setValue(int, int, double[])} method.
	 * 
	 * @return A new phentotype
	 */
	public abstract Phenotype createPhenotype();


	/**
	 * Creates a unique marking on the current branch.
	 * <p>
	 * This method is invoked during evolution when adding genes to the genome.
	 * 
	 * @return The marking
	 */
	public Marking createMarking();

	/**
	 * Creates a unique identifier on the current branch.
	 * <p>
	 * This method is invoked during evolution when adding genomes to a
	 * generation.
	 * 
	 * @return The identifier
	 */
	public Identifier createIdentifier();

	/**
	 * Creates an invalid marking.
	 * <p>
	 * This method is invoked to create a dummy marking so that it may be loaded
	 * from XML later.
	 * 
	 * @return The marking
	 */
	public Marking createInvalidMarking();

	/**
	 * Creates an invalid identifier.
	 * <p>
	 * This method is invoked to create a dummy identifier so that it may be loaded
	 * from XML later.
	 * 
	 * @return The marking
	 */
	public Identifier createInvalidIdentifier();

	/**
	 * Notifies the factory that the given marking exists already and may not
	 * be used when generating unique markings during evolution.
	 * 
	 * @param marking The marking
	 */
	public void reserveMarking(Marking marking);

	/**
	 * Notifies the factory that the given identifier exists already and may not
	 * be used when generating unique identifiers during evolution.
	 * 
	 * @param identifier The identifier
	 */
	public void reserveIdentifier(Identifier identifier);
}
