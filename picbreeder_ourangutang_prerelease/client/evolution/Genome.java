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

import client.*;
import java.util.*;

/**
 * A Genome contains a collection of genes responsible for constructing
 * a CPPN. There are two primary types of genes, explicitly available
 * through this interface, links and nodes.
 * <p>
 * Genomes should only be created by generators except for loading files
 * and initial populations. As genomes evolve, they may mutate genes.
 * When the phenome of a genome is required, a CPPN is constructed as
 * a snapshot of the genome at a particular time. The CPPN is then used
 * as a means to map the genome to the phenotype.
 * 
 * @author Nick Beato 
 */
public interface Genome extends Transferable, Cloneable, Comparable <Genome> {
	/**
	 * Gets all of the genes in this genome.
	 * <p>
	 * The links should be treated as "read only".
	 * 
	 * @return The genes
	 */
	public Collection<Gene> getGenes();
	
	/**
	 * Gets all of the links in this genome.
	 * <p>
	 * The links should be treated as "read only".
	 * 
	 * @return The links
	 */
	public Collection<Link> getLinks();
	
	/**
	 * Gets all of the nodes in this genome.
	 * <p>
	 * The nodes should be treated as "read only".
	 * 
	 * @return The nodes
	 */
	public Collection<Node> getNodes();
	
	/**
	 * Gets the generation number of this genome.
	 * 
	 * @return The generation of this genome 
	 */
	public long getAge();
	
	
	/**
	 * Returns the number of nodes in this genome.
	 * 
	 * @return The number of nodes
	 */
	public int countNodes();
	
	/**
	 * Returns the number of links in this genome.
	 * 
	 * @return The number of links
	 */
	public int countLinks();
	
	/**
	 * Randomly change all genes in this genome.  This is primarily here
	 * to initialize the root of a series.
	 */
	public void randomize();
	
	/**
	 * Provides a way to add a node to the genome.
	 * 
	 * @param node The node to add
	 */
	public void addNode(Node node);
	
	/**
	 * Provides a way to add a link to the genome.  Make sure the link is 
	 * not a duplicate!
	 * 
	 * @param link The link to add
	 */
	public void addLink(Link link);
	
	/**
	 * Checks if this genome already has a link between the given nodes.
	 * 
	 * @param source The source node
	 * @param destination The destination node
	 * @return <code>true</code> if a link connects these nodes already, <code>false</code> otherwise.
	 */
	public boolean hasLinkConnecting(Node source, Node destination);
	
	/**
	 * Gets the node with the specified marking.
	 * 
	 * @param marking The marking of the node
	 * @return The node if it exists, <code>null</code> otherwise.
	 */
	public Node getNode(Marking marking);
	
	/**
	 * Gets the link with the specified marking.
	 * 
	 * @param marking The marking of the link
	 * @return The link if it exists, <code>null</code> otherwise.
	 */
	public Link getLink(Marking marking);
	
	/**
	 * Adds a parent genome to this genome.
	 * <p>
	 * This method is invoked by the Series to keep track of ancestry.
	 * This allows the Series to prune the genomes based off of which
	 * genomes are useful.
	 * <p>
	 * It is also invoked by the Series when restoring a generation
	 * to rebind the genome after it has loaded.
	 * 
	 * @param parent
	 */
	public void addParent(Genome parent);
	
	/**
	 * Gets a list of the parents of this genome.
	 * <p>
	 * This method is invoked by the Genome when it must save
	 * itself.
	 * <p>
	 * It is also invoked by the Series when pruning the population.
	 * 
	 * @return The list of parents
	 */
	public Collection <Genome> getParents();
	
	/**
	 * Retreives the list of identifiers of parents of this genome.
	 * This method will only return the loaded parents.  It should
	 * only be invoked after generations have been loaded and the
	 * Series object needs to rebind the parents to the genome.
	 * 
	 * @return The list of parents' identifiers
	 */
	public Collection <Identifier> getParentIdentifiers();
	
	/**
	 * Gets this genome's identifier.
	 * 
	 * @return The identifier
	 */
	public Identifier getIdentifier();
}

