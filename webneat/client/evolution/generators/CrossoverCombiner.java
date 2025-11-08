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

package client.evolution.generators;

import client.evolution.*;
import java.util.*;
import client.utilities.Random;

/**
 * 2 Parent Crossover done the NEAT way. All genes from both
 * parents are combined into genome. If markings show that a
 * gene exists in both parents, the genes are "combined" as
 * best as possible. In this case, the activation is randomly
 * chosen and weights are averaged.
 */
public class CrossoverCombiner extends AbstractCombiner {
	public int minimumParents() {
		return 2;
	}
	
	public Genome generate(Collection <Genome> parents) {
		LinkedList <Genome> ps = new LinkedList <Genome> ();
		ps.addAll(parents);
		
		Genome dad = ps.get(Random.instance().nextInt(ps.size()));
		ps.remove(dad);
		
		Genome mom = ps.get(Random.instance().nextInt(ps.size()));
		ps.clear();
		
		Genome kid = GeneticFactoryInstance.get().copyGenome(mom);
		
		// lightweight maintanence
		kid.addParent(mom);
		kid.addParent(dad);
		
		// add all nodes to the kid from both parents
		// copy automatically moved all from the mom to the kid
		for(Node node : dad.getNodes()) {
			Node k = kid.getNode(node.getMarking());
			
			// kid does not have the dad's gene, so add it
			if(k == null)
				kid.addNode(GeneticFactoryInstance.get().copyNode(node));
			
			// kid does have the dad's gene, so randomly choose if mom or 
			// dad propogates the activation for it
			else if(Random.instance().nextBoolean())
				k.setActivation(node.getActivation());
		}
		
		// add all links to the kid. again all of the mom's links exist.
		for(Link link : dad.getLinks()) {
			Link k = kid.getLink(link.getMarking());
			
			// kid does not have the dad's gene, so just add it
			if(k == null)
				kid.addLink(GeneticFactoryInstance.get().copyLink(link));
			
			// if a link exists in both parents, average the weight
			else
				k.setWeight((link.getWeight() + k.getWeight()) / 2.0);
		}
		
		// should not have to sort the genes since only addLink and addNodes
		// modified the genome. everything should be ok now
		return kid;
	}
}
